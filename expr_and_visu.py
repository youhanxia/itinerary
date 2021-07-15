import cPickle as pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.stats import ttest_ind, ttest_rel
import csv


def read_in():
    # data = dict()
    # for fn in os.listdir('results'):
    #     with open('results/' + fn) as f:
    #         d = pickle.load(f)
    #     ind = fn[:-4].split('_')
    #     if ind[0] not in data:
    #         data[ind[0]] = dict()
    #     if ind[1] not in data[ind[0]]:
    #         data[ind[0]][ind[1]] = dict()
    #     if ind[2] not in data[ind[0]][ind[1]]:
    #         data[ind[0]][ind[1]][ind[2]] = dict()
    #     data[ind[0]][ind[1]][ind[2]][ind[3]] = d
    # with open('results/results.txt', 'w') as f:
    #     pickle.dump(data, f)
    with open('results/results.txt') as f:
        data = pickle.load(f)
    return data


def acs_comp(data, indices):
    comp = dict()
    plt.title('single obj meal time comparison')
    plt.ylabel('meal time difference (h)')
    for par in indices['pars_acs']:
        comp[par] = []
        for se in indices['ses']:
            comp[par].append(np.mean([abs(ele[0] - 13.0) for ele in data['acs'][par][se]['1.0']['meal_times']]))
        plt.plot(comp[par], label=par)
        plt.legend()
    plt.show()


def acs_comp_scatter(data, indices):
    colors = {'b': 'b.', 'g': 'g.', 'gl': 'r.'}
    comp = dict()
    plt.title('single obj meal time comparison')
    plt.ylabel('meal time difference (h)')
    for par in ['b', 'g', 'gl']:
        comp[par] = []
        for i in xrange(30):
            comp[par].extend([(i, abs(item - 13.0)) for sublist in data['acs'][par][indices['ses'][i]]['1.0']['meal_times'] for item in sublist])
        xs, ys = zip(*comp[par])
        plt.plot(xs, ys, colors[par], label=par)
    plt.legend()
    plt.show()


def macs_front(data, indices):
    par = 'b'

    fig = plt.figure()
    ax = plt.axes(xlim=(0, 25000), ylim=(-0.1, 1), xlabel='utility', ylabel='meal time difference (h)')
    front, = ax.plot([], [], '.')
    plt.title('')

    def animate(i):
        j = i / 30
        k = i % 30
        plt.title('pareto front of instance ' + indices['ses'][j])
        front.set_xdata(data['macs'][par][indices['ses'][j]]['1.0']['utilities'][k])
        front.set_ydata([abs(13.0 - ele) for ele in data['macs'][par][indices['ses'][j]]['1.0']['meal_times'][k]])

    ani = animation.FuncAnimation(fig, animate, np.arange(900), repeat=True, interval=100)
    plt.show()


def acs_tour(data, indices):
    par = 'g'

    poi_loc = dict()
    with open('POI-LA-top100.csv') as f:
        for l in csv.DictReader(f, delimiter=';'):
            poi_loc[l['poiID']] = (float(l['lat']), float(l['long']))

    fig = plt.figure()
    ax = plt.axes(xlim=(33, 35), ylim=(-119, -118))
    line, = ax.plot([], [])
    plt.title('')

    def init():
        lats, longs = zip(*poi_loc.values())
        plt.scatter(lats, longs)

    def animate(i):
        j = i / 30
        k = i % 30
        plt.title('tour on map of instance ' + indices['ses'][j])
        tour = data['acs'][par][indices['ses'][j]]['1.0']['summary'][k][0]['tour']
        pois = [u for u, _ in tour]
        pois.append(tour[-1][1])
        locs = [poi_loc[str(poi)] for poi in pois]
        line.set_data(zip(*locs))

    ani = animation.FuncAnimation(fig, animate, np.arange(900), init_func=init, repeat=False, interval=100)
    # init()
    plt.show()


def macs_comp_by_se(data, indices):
    for se in indices['ses']:
        plt.clf()
        plt.title('solutions of ' + se)
        plt.xlabel('utility')
        plt.ylabel('meal time difference (h)')
        xs_b = []
        ys_b = []
        xs_bg = []
        ys_bg = []
        for i in xrange(30):
            xs_b.extend(data['macs']['b'][se]['1.0']['utilities'][i])
            ys_b.extend(abs(13.0 - t) for t in data['macs']['b'][se]['1.0']['meal_times'][i])
            xs_bg.extend(data['macs']['bg'][se]['1.0']['utilities'][i])
            ys_bg.extend(abs(13.0 - t) for t in data['macs']['bg'][se]['1.0']['meal_times'][i])
        plt.plot(xs_b, ys_b, '.')
        plt.plot(xs_bg, ys_bg, '.g')
        # plt.show()
        plt.savefig('imgs/macs_front_comp_' + se)


def acs_in_macs(data, indices):
    colors = {'b': 'b.', 'g': 'g.', 'gl': 'r.'}
    for se in indices['ses']:
        plt.clf()
        plt.title('SO in MO' + se)
        plt.xlabel('utility')
        plt.ylabel('meal time difference (h)')
        xs = []
        ys = []
        for i in xrange(30):
            xs.extend(data['macs']['b'][se]['1.0']['utilities'][i])
            ys.extend(abs(13.0 - t) for t in data['macs']['b'][se]['1.0']['meal_times'][i])
        plt.plot(xs, ys, 'y.')
        for par in indices['pars_acs']:
            xs = [data['acs'][par][se]['1.0']['utilities'][i][0] for i in xrange(30)]
            ys = [abs(13.0 - data['acs'][par][se]['1.0']['meal_times'][i][0]) for i in xrange(30)]
            plt.plot(xs, ys, colors[par])
        # plt.show()
        plt.savefig('imgs/so_in_mo' + se)


def acs_on_diff_dur(data, indices):
    se_0 = indices['ses'][9]
    plt.figure(figsize=(15, 6))
    y1 = dict()
    y2 = dict()
    for dur in ['0.5', '1.0', '2.0']:
        y1[dur] = []
        # y2[dur] = []
        for se in indices['ses']:
            y1[dur].append(np.mean([abs(ele[0] - 13.0) for ele in data['acs']['b'][se][dur]['meal_times']]))
            # y2[dur].append(np.mean(data['acs']['b'][se][dur]['utilities']))
            if se == se_0:
                y2[dur] = [abs(ele[0] - 13.0) for ele in data['acs']['b'][se][dur]['meal_times']]
        # plt.subplot(221)
        # plt.title('meal time')
        # plt.plot(y1[dur], label=dur)
        # plt.legend(loc='best', framealpha=0.3)
        # plt.subplot(222)
        # plt.title('utility')
        # plt.plot(y2[dur], label=dur)
        # plt.legend(loc='best', framealpha=0.3)

    plt.subplot(121)
    plt.title('meal time error at ' + se_0)
    plt.xlabel('duration factor')
    plt.ylabel('meal time error (h)')
    plt.boxplot([y2['0.5'], y2['1.0'], y2['2.0']], labels=['0.5', '1.0', '2.0'])
    # plt.subplot(223)
    plt.subplot(122)
    plt.title('meal time error average of all instances (h)')
    plt.xlabel('duration factor')
    plt.ylabel('meal time error (h)')
    plt.boxplot([y1['0.5'], y1['1.0'], y1['2.0']], labels=['0.5', '1.0', '2.0'])
    # plt.subplot(224)
    # plt.subplot(122)
    # plt.title('utility')
    # plt.boxplot([y2['0.5'], y2['1.0'], y2['2.0']], labels=['0.5', '1.0', '2.0'])
    # plt.show()
    plt.savefig('imgs/baseline_comp_on_dur')


def acs_table(data, indices):

    title = {'meal_times': 'meal time error', 'meal_ratings': 'meal rating', 'utilities': 'utility'}
    prec = '& %.3f'
    
    def meal_time(data):
        return [abs(ele[0] - 13.0) for ele in data['meal_times']]

    dur = '1.0'
    for prop in indices['result_contents'][:-1]:
        print '\\begin{table}'
        print '\\caption{baseline vs. S-ItiMeal on ' + title[prop] + '}'
        # print '\\begin{adjustbox}{width=1\\textwidth}'
        print '\\begin{tabular}{l|rrrr|r}'
        print '& \\multicolumn{2}{c}{baseline} & \multicolumn{2}{c}{S-ItiMeal} & \\\\'
        print 'Instance & Mean & Std & Mean & Std & Significance \\\\'
        print '\\hline'
        for se in indices['ses']:
            data1 = data['acs']['b'][se][dur]
            data2 = data['acs']['g'][se][dur]
            if prop == 'meal_times':
                print se, prec % np.mean(meal_time(data1)), prec % np.std(meal_time(data1)), prec % np.mean(
                    meal_time(data2)), prec % np.std(meal_time(data2)), '&',
                pv = ttest_ind(meal_time(data1), meal_time(data2), equal_var=False, nan_policy='omit')[1] if meal_time(data1) != meal_time(data2) else 1.0
                if pv <= 0.0001:
                    sym = '****'
                elif pv <= 0.001:
                    sym = '***'
                elif pv <= 0.01:
                    sym = '***'
                elif pv <= 0.05:
                    sym = '***'
                else:
                    sym = ''
                print sym, '\\\\'
            else:
                print se, prec % np.mean(data1[prop]), prec % np.std(data1[prop]), prec % np.mean(
                    data2[prop]), prec % np.std(data2[prop]), '&',
                pv = ttest_ind(data1[prop], data2[prop], equal_var=False)[1][0] if data1[prop] != data2[prop] else 1.0
                if pv <= 0.0001:
                    sym = '****'
                elif pv <= 0.001:
                    sym = '***'
                elif pv <= 0.01:
                    sym = '***'
                elif pv <= 0.05:
                    sym = '***'
                else:
                    sym = ''
                print sym, '\\\\'
        print '\\end{tabular}'
        # print '\\end{adjustbox}'
        print '\\end{table}'
        print


def front_vs_single(data, ses):
    j = 0
    for se in ses:
        for i in [28]:  # xrange(30):
            front_x = data['macs']['b'][se]['1.0']['utilities'][i]
            point_x = data['acs']['g'][se]['1.0']['utilities'][i][0]
            base_x = data['acs']['b'][se]['1.0']['utilities'][j][0]
            front_y = [abs(13.0 - t) for t in data['macs']['b'][se]['1.0']['meal_times'][i]]
            point_y = abs(13.0 - data['acs']['g'][se]['1.0']['meal_times'][i][0])
            base_y = abs(13.0 - data['acs']['b'][se]['1.0']['meal_times'][j][0])
            front_z = data['macs']['b'][se]['1.0']['meal_ratings'][i]
            # plt.figure(figsize=(10, 10))
            plt.plot(front_x, front_y, '.', label='M-ItiMeal', color='b')
            front_xyz= zip(front_x, front_y, front_z)
            ratings = set(map(lambda x: x[2], front_xyz))
            xys = [([(xyz[0], xyz[1]) for xyz in front_xyz if xyz[2] == r], r) for r in ratings]
            for xy, r in xys:
                xy.sort()
                x, y = zip(*xy)
                if r == 5.0:
                    s = '--'
                elif r == 4.0:
                    s = '-.'
                else:
                    s = '-'
                plt.plot(x, y, s, label='rating='+str(r), color='k')
            plt.plot(point_x, point_y, 'o', label='S-ItiMeal', color='r')
            plt.annotate('S-ItiMeal', (point_x, point_y), textcoords='data')
            plt.plot(base_x, base_y, '^', label='Greedy POI', color='g')
            plt.annotate('Greedy POI', (base_x, base_y), textcoords='data')
            # plt.title('pareto front from M-ItiMeal compared with solution of S-ItiMeal')
            # plt.title(se)
            plt.legend(loc='best')
            plt.xlabel('utility')
            plt.ylabel('meal time error (h)')
            plt.gca().invert_yaxis()
            axes = plt.gca()
            boarder = 0.1
            xmin, xmax = axes.get_xlim()
            ymax, ymin = axes.get_ylim()
            xmin = xmin - (xmax - xmin) * boarder
            xmax = xmax + (xmax - xmin) * boarder
            ymin = ymin - (ymax - ymin) * boarder
            ymax = ymax + (ymax - ymin) * boarder
            axes.set_xlim([xmin, xmax])
            axes.set_ylim([ymax, ymin])
            # plt.show()
            # plt.savefig('imgs/front_vs_single')


def front_vs_single_2by2(data, ses):
    plt.figure(figsize=(10, 10))
    plt.subplot(221)
    front_vs_single(data, [ses[0]])
    plt.subplot(222)
    front_vs_single(data, [ses[1]])
    plt.subplot(223)
    front_vs_single(data, [ses[2]])
    plt.subplot(224)
    front_vs_single(data, [ses[3]])
    # plt.show()
    plt.savefig('imgs/front_vs_single_2by2')

def ratio(data, indices):
    y1 = []
    y2 = []
    for se in indices['ses']:
        data1 = data['acs']['b'][se]['1.0']
        data2 = data['acs']['g'][se]['1.0']
        y1.append(np.mean([ele[0] for ele in data2['utilities']]) / np.mean([ele[0] for ele in data1['utilities']]))
        y2.append(np.mean([abs(ele[0] - 13.0) for ele in data2['meal_times']]) / np.mean([abs(ele[0] - 13.0) for ele in data1['meal_times']]))

    ind = np.arange(30)
    width = 0.35
    plt.figure(figsize=(15, 6))
    # plt.title('Comparison between S-ItiMeal and Baseline on 30 start-end pairs')
    plt.xlabel('Experiment Index')
    plt.ylabel('Ratio of S-ItiMeal to Greedy POI')
    plt.bar(ind, y1, width, color='b', edgecolor='none', label='Utility')
    plt.bar(ind + width, y2, width, color='r', edgecolor='none', label='Meal time error')
    lgd = plt .legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # plt.show()
    plt.savefig('imgs/ratio', bbox_extra_artists=(lgd,), bbox_inches='tight')


def ratio_scatter(data, indices):
    x = []
    y = []
    for se in indices['ses']:
        data1 = data['acs']['b'][se]['1.0']
        data2 = data['acs']['g'][se]['1.0']
        x.append(np.mean([ele[0] for ele in data2['utilities']]) / np.mean([ele[0] for ele in data1['utilities']]))
        y.append(np.mean([abs(ele[0] - 13.0) for ele in data2['meal_times']]) / np.mean(
            [abs(ele[0] - 13.0) for ele in data1['meal_times']]))

    plt.figure(figsize=(10, 10))
    plt.xlabel('S-ItiMeal_Utility / Greedy_POI_Utility')
    plt.ylabel('S-ItiMeal_Meal_Time_Error / Greedy_POI_Meal_Time_Error')
    plt.plot(x, y, '+', color='k', label='Single start end pair')
    plt.plot(np.mean(x), np.mean(y), 'o', color='b', label='Average')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.legend()
    # plt.show()
    plt.savefig('imgs/ratio_scatter')
    # print np.mean(x), np.mean(y)


def main():
    indices = dict()
    indices['modes'] = ['acs', 'macs']
    indices['pars_acs'] = ['b', 'g', 'gl']
    indices['pars_macs'] = ['b', 'bg']
    indices['ses'] = ['(155, 154)', '(232, 124)', '(141, 387)', '(55, 157)', '(378, 137)', '(337, 273)', '(295, 188)',
                      '(39, 272)', '(219, 55)', '(264, 182)', '(271, 328)', '(73, 102)', '(150, 389)', '(15, 76)',
                      '(339, 390)', '(202, 391)', '(41, 304)', '(414, 26)', '(294, 220)', '(157, 31)', '(321, 167)',
                      '(390, 141)', '(286, 394)', '(260, 388)', '(76, 293)', '(99, 39)', '(137, 321)', '(394, 260)',
                      '(10, 155)', '(309, 150)']
    indices['dur_factors'] = ['0.5', '1.0', '2.0']
    indices['result_contents'] = ['meal_times', 'utilities', 'meal_ratings', 'summary']

    data = read_in()
    # acs_comp(data, indices)
    # acs_comp_scatter(data, indices)
    # macs_front(data, indices)
    # acs_tour(data, indices)
    # macs_comp_by_se(data, indices)
    # acs_in_macs(data, indices)
    # acs_on_diff_dur(data, indices)
    # acs_table(data, indices)
    # front_vs_single(data, indices['ses'])
    # front_vs_single(data, ['(141, 387)'])
    # front_vs_single_2by2(data, ['(141, 387)', '(264, 182)', '(271, 328)', '(99, 39)'])
    # ratio(data, indices)
    ratio_scatter(data, indices)
    # print len(data['macs']['b']['(10, 155)']['1.0']['utilities'])

if __name__ == '__main__':
    main()
