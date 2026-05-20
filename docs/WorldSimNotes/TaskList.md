# Table of Contents

# Initial Vibe Code

# Dev Modules
## First runs of the sim
1. [x] Figure out how the vibe coded shit actually works
2. [x] clean things up so that it actually works as intended
3. [x] Get the 500 tick sim to work so the colony doesn't immediately die
After figuring out how to use the vibe coded stuff I ran some tests and debugged a couple things. The sim runs for 1000 ticks on the IDLE directive with no problems.
4. [x] Need to expand the logic from the IDLE directive to the other directives
5. [x] Testing with IDLE led to a whole rework of the directive system since directives need to be more flexible. It works on IDLE but as soon as I switch to anything else it dies
6. Need to make a default logic loop that acts like IDLE but is more to just make sure the colony survives
7. I also need a more flexible directive system
## Directive Implementation
- Did a lot of work to get a new directive system up an running. Will be stored in its own file so that things are a little more self contained. Now the directives specify a target resource and urgency
	- Based on the resource given the directive there is a priority list for what the colony should be building.
	- If there are any critical flags those get addressed first
	- More work on this to get the directives fully fleshed out.
		- Before executing the directive a basic survival loop needs to be passed to make sure the colony wont die.
		- Once that is passed the directive is followed
		- Fixed some
### Other Ideas:
- Rebellions ==(will be implemented later)==
	- If a colony is unhappy enough it will rebel against the faction. when a colony rebels it causes nearby colonies to become more unhappy because now they have an enemy very close to them and potentially lose a trade partner
	- If the happiness of a nearby colony drops below the rebellion threshold then it will also rebel causing another wave of unhappiness to nearby colonies
	- This could create a cascading effect of lots of nearby colonies rebelling against the faction
1. [ ] This forced some thought about trade between colonies
	1. [ ] Storing this information is difficult and gets even more complex when there are colonies that begin rebelling
	2. [ ] Might want to disable this for now and then come back once the whole sim works and do a "trade update"
	3. [ ] ==Colony trade will definitely be later.== Need a colony to be able to send resources to a new system so I need to completely flesh out the colony ships and make sure they are feasible for a successful colony.
### Final Notes
#### Completed Tasks:
- There is both a directive type and target resource
	- The directive type tells the colony what actions to take while the target resource tells it what resource it should be applying this logic to.
- Fixed a bug with assigning and unassigning workers to buildings
- Fixed a bug with buildings not going into repair mode and then also letting themselves die before going inactive.
- Defense system is kind of implemented. The colony will grow the resource until a defense per person value is met and then it will unstaff all workers from forts.
	- I think this defense resource can be used against other colonies which would cause it to decrease.
	- If you lose against another colony that's just resources down the drain. Might be good to dedicate a number of people that have to be present per unit of defense. Maybe it could be a research upgrade where defensive units always cost 1 person but have greater amounts of impact on the defense score
#### Persisting issues:
- Sim run for 500 ticks still shows issues
	- This is false, we start to see problems emerge at 400 ticks. Need to investigate what's causing that
	- Sim gets in a weird loop of building farms, then building power plants, then building forts to support the growing population.
	- Not sure what's really causing it but it has to be related to the basic survival loop.
		- I can add some print statements to basic survival loop to give more insight about where the logic is failing there.
		- It could have something to do with buildings going down for maintenance all at the same time and causing a sudden drop in organics/power rate.
	- Also need to fix the repairing line on the building plot function in snaspshot.py
- Next time I'll get it to expand into a new system
	- Also makes me realize that I might need to rework the planet system and natural resources since the buildings don't actually take all the planets resources, they just make them. 
	- I could have a colony building start with a default resource rate that is above sustainable so that new buildings can be made
	- This would mean I also need to start making buildings to increase population, nah they just live where they work
- I should also get this more up to date so I have stuff to reference when I'm away for a week or two. Its hard to relearn everything
- Whatever I end up actually working on, I'm done for today. POE2 time

## New Colony Implementation 5/20/26
Starting a list of things that need done to get the colony to make a new colony and add to the faction.
Figure out why the colony gets into weird logic loops of making buildings beyond the point that it needs to
Once its stable and can make lots of resources, make it focus on creating a new colony ship and sending resources there to start a new place.
==Make a new building called outpost or capital building that will passively generate resources to allow a colony to start itself off and not need other buildings.==


# List of future implementations
## Colony Trade
Write my notes here on how to implement it
## Rebellions
